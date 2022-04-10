// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ManageOwner {
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function setOwner() internal {
        owner = msg.sender;
    }

    function changeOwner(address _newOwner) public onlyOwner {
        owner = _newOwner;
    }
}

contract ManageContract is ManageOwner {
    bool public stopped = false;

    modifier isStopped() {
        require(!stopped);
        _;
    }

    function stopContract() public onlyOwner isStopped {
        stopped = true;
    }

    function resumeContract() public onlyOwner {
        require(stopped);
        stopped = false;
    }

    function destroyContract() public payable onlyOwner {
        selfdestruct(payable(owner));
    }
}

contract SimpleFaucet is ManageContract {
    mapping(address => uint256) public donationRecoed;
    uint256 public anonymousDonation;

    event Received(address, uint256);
    event Fallbacked(address, uint256);

    constructor() {
        setOwner();
    }

    receive() external payable {
        emit Received(msg.sender, msg.value);
        anonymousDonation += msg.value;
    }

    fallback() external payable {
        emit Fallbacked(msg.sender, msg.value);
        revert();
    }

    function getFaucetBalance() public view returns (uint256) {
        return address(this).balance;
    }

    function donateEther() public payable {
        donationRecoed[msg.sender] += msg.value;
    }

    function sendEther(uint256 _amount) public payable {
        require(_amount <= address(this).balance);

        payable(msg.sender).transfer(_amount);
    }
}
